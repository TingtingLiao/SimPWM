"""Proxy runner for DROID episodes: first ext1 frame + logged robot states + camera -> proxy video, evaluated against
the real video.

Camera, per episode (first that applies): CAMERA_DIR/<idx>.json fitted on the episode's recording
(droid/episode_calib.py), the camera DROID logged (`--logged-camera`, ablation), else the pre-calibrated camera of its
setup (droid/setup_calib.py, fitted on OTHER episodes of the setup). First-frame metric depth: stereo
(STEREO_DIR/<idx>_ext1.npy from droid/stereo/extract.py; episodes without one are skipped so comparisons stay paired),
else monocular.

Outputs OUT_DIR/droid_XXXXX/: check.mp4 (real | overlay / depth | categories), conditions/ (proxy depth + semantics,
24 fps, 192x336; with `--ref-size WxH` also the LTX control videos ref_WxH.mp4 and the RGBD proxy ref_rgb_WxH.mp4 +
ref_depth_WxH.mp4: the first CLIP_FRAMES recorded steps at 15 fps, rendered at the raw recording resolution),
metrics.json (+ metrics_sam3.json), proxy_cat.npz, first_frame_src.png, prompt.txt, info.json (written last: finished
episodes are skipped).

`--hires` (preset "droid_hires", simpwm/config.py; needs `--ref-size` and the raw DROID source): the simulation runs at
640x360, and SAM 3, the object masks, colours and textures and the background depth of the control videos use the
full-resolution first frame.

usage: python -m simpwm.droid.run OUT_DIR IDX [IDX ...] [--camera-dir DIR] [--stereo-dir DIR] [--ref-size 1280x704]
                               [--clip-frames 97] [--clip-only] [--setups JSON] [--hires]
"""

import argparse
import colorsys
import json
import traceback
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image

from simpwm import config, paths
from simpwm.core.camera import Camera
from simpwm.core.embodiment import Embodiment
from simpwm.core.io import write_mp4
from simpwm.droid import data, setup_calib
from simpwm.eval import tracks as evaluate
from simpwm.sim import camera_calib, init_state, pipeline, scale
from simpwm.sim import render as raster

CAT_PAL = np.array([[70, 70, 70], [150, 120, 80], [230, 230, 230], [255, 160, 0], [255, 0, 0]] +
                   [[int(c) for c in np.array(colorsys.hsv_to_rgb(h, 0.8, 1.0)) * 255]
                    for h in (0.3 + 0.618034 * np.arange(24)) % 1.0], np.uint8)  # golden-ratio hues: neighbours differ
CWM_FPS, CWM_T, CWM_HW = 24, 124, (192, 336)  # conditions/ frames: rate, count, size
RAW_HW = (720, 1280)  # DROID raw ext1 recording size


def contact_vis(cat, con):
    """Category colours stay fixed per object; contact is drawn as a yellow outline of the touched region (filling it
    would change the held object's colour while it is held)."""
    im = CAT_PAL[np.clip(cat, 0, len(CAT_PAL) - 1)].copy()
    m = (con >= 0.5).astype(np.uint8)
    if m.any():
        edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
        im[edge] = (255, 255, 0)
    return im


def depth_vis(d, lo=0.2, hi=2.5):
    v = np.where(d > 0, 1 - (np.clip(d, lo, hi) - lo) / (hi - lo), 0)
    return (np.stack([v, v, v], -1) * 255).astype(np.uint8)


def _footprint(cam, pts):
    """Image footprint of a point set without occlusion (filled convex hull of its projection)."""
    uv, z = cam.project(pts)
    m = np.zeros((cam.H, cam.W), np.uint8)
    uv = uv[z > 0]
    if len(uv) >= 3:
        cv2.fillConvexPoly(m, cv2.convexHull(np.round(uv).astype(np.int32)), 1)
    return m.astype(bool)


def _camera(idx, ep, emb, others, ok, camera_dir):
    """(camera, calibration report) of the episode; sets the embodiment's gripper yaw."""
    cf = Path(camera_dir) / f"{idx}.json" if camera_dir else None
    if cf is not None and cf.exists():  # fitted on this recording (droid/episode_calib.py)
        c = json.loads(cf.read_text())
        cam = Camera(np.array(c["K"]), np.array(c["T_wc"]), ep["cam"].H, ep["cam"].W, tuple(c["dist"]))
        if config.cfg.hires and (ep["cam"].H, ep["cam"].W) != (180, 320):
            # the per-episode cameras were fitted at 320x180 (droid/episode_calib.py): same camera, sim resolution
            cam = Camera(np.array(c["K"]), np.array(c["T_wc"]), 180, 320, tuple(c["dist"])).scaled(ep["cam"].H,
                                                                                                   ep["cam"].W)
        emb.set_gripper_yaw(c["gripper_yaw"])
        return cam, dict(gripper_yaw=c["gripper_yaw"], episode_refine=dict(start=c["start"], iou_before=c["iou_before"],
                                                                           iou_after=c["iou_after"]))
    if config.cfg.hires:  # the setup cameras are calibrated and cached at 320x180
        raise RuntimeError(f"hires: no per-episode camera of {idx} (--camera-dir resources/droid/cameras_hires)")
    if config.cfg.logged_camera:  # ablation: DROID's own calibration as recorded, no fitting
        return ep["cam"], dict(logged=True)
    if others is None:
        others = setup_calib.same_setup(idx)
    return setup_calib.episode_camera(idx, others, ok, emb, ep["cam"])


def label_held(res):
    """Keep only the robot and the objects the effector actually held as labels in the proxy frames (in place); the
    support and the other objects stay in the depth map but are labelled background (they are already in the first
    frame). Returns the ids of the held objects."""
    held = sorted({int(k) for s in res["states"] for k, v in s["contacts"].items() if v >= 1.0})

    def shown(cat):
        obj = cat >= raster.C_OBJ0
        keep = np.isin(cat, [raster.C_OBJ0 + i for i in held])
        return np.where((obj & ~keep) | (cat == raster.C_SUPPORT), raster.C_BG, cat).astype(cat.dtype)

    res["frames"] = [(d, shown(c), con) for d, c, con in res["frames"]]
    res["frames_hi"] = {t: (d, shown(c), con) for t, (d, c, con) in res["frames_hi"].items()}
    res["info"]["held_objects"] = held
    return held


def run(idx, out_dir, others, ok, camera_dir=None, ref_size=None, clip_frames=97, clip_only=False,
        vlm_dir=paths.RUNS / "vlm_objects", edit=None):
    """Simulate, evaluate and export episode idx (module docstring).

    others: the other episodes of its setup (setup calibration input; None = from the setup index). ref_size "WxH":
    render the LTX control videos of the first clip_frames steps. clip_only: simulate only those steps and skip the
    evaluation (object tracks, robot IoU, SAM 3 tracker; metrics.json then holds only the proxy's own object entries).
    edit(idx, ep) -> ep: optional change of the loaded episode (droid/run_counterfactual.py).
    """
    ep = data.load(ok[idx])
    if edit is not None:
        ep = edit(idx, ep)
    if clip_only:
        T0 = len(ep["frames"])
        cut = lambda v: v[:clip_frames] if isinstance(v, np.ndarray) and len(v) == T0 else v  # noqa: E731
        ep = {k: ({kk: cut(vv) for kk, vv in v.items()} if k == "meta" else cut(v)) for k, v in ep.items()}
    out = Path(out_dir) / f"droid_{idx:05d}"
    out.mkdir(parents=True, exist_ok=True)
    emb = Embodiment("franka_robotiq")
    cam, cal = _camera(idx, ep, emb, others, ok, camera_dir)
    print("camera calibration:", {k: v for k, v in cal.items() if k != "x"}, flush=True)
    soft_px = ()
    vf = Path(vlm_dir) / f"{idx}.json"
    if config.cfg.soft_objects and vf.exists():  # closing points the VLM named deformable (perception/vlm_objects.py)
        soft_px = [tuple(g["px"]) for g in json.loads(vf.read_text())["grasps"] if pipeline.is_soft(g["phrase"])]
        if config.cfg.hires:  # the VLM closing points are 320x180 pixels
            soft_px = [(int(round(u * scale.s())), int(round(v * scale.s()))) for u, v in soft_px]
    # the first clip_frames recorded steps (native 15 fps, no resampling) are also rendered with the camera at the raw
    # recording resolution: the LTX control videos
    ref_t = hires = None
    if ref_size:
        ref_t = np.minimum(np.arange(clip_frames), len(ep["frames"]) - 1)
        hires = (cam.scaled(*RAW_HW), set(ref_t.tolist()))
    frame0_hi = hires_info = None
    if config.cfg.hires:  # full-resolution first frame: SAM 3, object colours / textures, RGB proxy background
        if hires is None:
            raise RuntimeError("--hires needs --ref-size (the full-resolution proxy render)")
        from simpwm.video.dataset import raw_first_frame

        frame0_hi, hires_info = raw_first_frame(idx, ep)
    res = pipeline.generate(ep["frames"][0], ep["arm_q"], ep["grip"], cam, emb, ep["fps"], refine_cam=False,
                            soft_px=soft_px, hires=hires, frame0_hi=frame0_hi)
    if hires_info is not None:
        res["info"]["hires"] = dict(res["info"].get("hires", {}), **hires_info)
    res["info"]["camera_calibration"] = cal
    T = len(ep["frames"])
    m0 = {o.id: m for o, m in zip(res["objects0"], res["dbg"]["obj_masks"])}
    tracked = evaluate.track(ep["frames"], m0) if m0 and not clip_only else {}
    metrics = evaluate.compare(res["frames"], tracked)
    # tracking reliability and proxy/real disagreement, so a lost track is not counted as a proxy error
    held_ids = {int(k) for s in res["states"] for k, v in s["contacts"].items() if v >= 1.0}
    for k, r in metrics.items():
        sc = getattr(evaluate.track, "scores", {}).get(int(k)) if tracked else None
        if sc is not None:
            r["track_score_min"] = float(np.nanmin(sc))
            r["track_absent_frac"] = float(np.nanmean(sc < 0))
        r["proxy_held"] = int(k) in held_ids
        r["disagree"] = bool(r["proxy_held"] != (r["moved_px"] >= 8))
    # robot: re-detect + segment the robot every few frames (tracking from frame 0 fails when it starts off-screen);
    # clip-only runs skip it (the camera was already scored on the clip by droid/episode_calib.py)
    every = [] if clip_only else list(range(0, T, max(1, T // 12)))
    rm = camera_calib.robot_masks(ep["frames"], emb, ep, every, res["cam"], source="sam2") if every else {}
    rious = {k: float((raster.is_robot(res["frames"][k][1]) & m).sum()
                      / max(1, (raster.is_robot(res["frames"][k][1]) | m).sum())) for k, m in rm.items()}
    r3 = camera_calib.robot_masks(ep["frames"], emb, ep, every, res["cam"], source="sam3") if every else {}
    i3, ic = [], []
    for k, m in rm.items():
        if k in r3:
            r = raster.is_robot(res["frames"][k][1])
            i3.append(float((r & r3[k]).sum() / max(1, (r | r3[k]).sum())))
            ic.append(float((r & m & r3[k]).sum() / max(1, (r | (m & r3[k])).sum())))
    metrics["robot"] = dict(iou=float(np.mean(list(rious.values()))) if rious else None, frames=rious,
                            iou_sam3=float(np.mean(i3)) if i3 else None, iou_cons=float(np.mean(ic)) if ic else None)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    # proxy object labels per frame (all objects, before display filtering) so metrics can be recomputed later;
    # objects without a first-frame segment (phantoms added at closures, sim/refine.phantom_grasps): their projected
    # frame-0 footprint, without occlusion, stands in for the t0 mask (they are not tracked)
    ph = [o for o in res["objects0"] if o.id not in m0]
    none = np.zeros((0,) + ep["frames"].shape[1:3], bool)
    np.savez_compressed(out / "proxy_cat.npz", cat=np.stack([c for _, c, _ in res["frames"]]).astype(np.int16),
                        masks0=np.stack([m0[k] for k in sorted(m0)]) if m0 else none,
                        obj_ids=np.array(sorted(m0), np.int32),
                        phantom_masks0=np.stack([_footprint(res["cam"], o.pts) for o in ph]) if ph else none,
                        phantom_ids=np.array([o.id for o in ph], np.int32))
    if m0 and not clip_only:  # independent evaluation reference: SAM 3 tracker
        from simpwm.eval import tracks_sam3 as evaluate_sam3

        tr3 = evaluate_sam3.track(ep["frames"], m0)
        m3 = evaluate.compare(res["frames"], tr3)
        for k, r in m3.items():
            sc = evaluate_sam3.track.scores.get(int(k))
            if sc is not None:
                r["track_absent_frac"] = float(np.nanmean(sc < 0))
            r["proxy_held"] = int(k) in held_ids
        (out / "metrics_sam3.json").write_text(json.dumps(m3, indent=1))
    print(json.dumps({k: {kk: v for kk, v in r.items() if kk not in ("iou_curve", "frames")}
                      for k, r in metrics.items()}), flush=True)
    held = label_held(res)
    vid = []
    for t in range(T):
        d, cat, con = res["frames"][t]
        real = ep["frames"][t]
        ov = real.copy()
        m = cat >= raster.C_ARM
        ov[m] = (0.5 * ov[m] + 0.5 * CAT_PAL[np.clip(cat[m], 0, len(CAT_PAL) - 1)]).astype(np.uint8)
        for oid, tr in tracked.items():
            cs, _ = cv2.findContours(tr[t].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(ov, cs, -1, (0, 255, 0), 1)
        vid.append(np.concatenate([np.concatenate([real, ov], 1),
                                   np.concatenate([depth_vis(d), contact_vis(cat, con)], 1)], 0))
    write_mp4(vid, out / "check.mp4", fps=ep["fps"])
    # proxy conditioning frames: resampled to 24 fps, the first CWM_T frames, CWM_HW
    cond = out / "conditions"
    cond.mkdir(exist_ok=True)
    src_t = np.minimum(np.round(np.arange(CWM_T) * ep["fps"] / CWM_FPS).astype(int), T - 1)
    for k, t in enumerate(src_t):
        d, cat, _ = res["frames"][t]
        dz = np.array(Image.fromarray(d).resize(CWM_HW[::-1], Image.NEAREST), np.float32)
        sem = np.array(Image.fromarray(raster.to_cwm(cat)).resize(CWM_HW[::-1], Image.NEAREST))
        dz.tofile(cond / f"{k:06d}.depth.f32")
        Image.fromarray(sem, mode="L").save(cond / f"{k:06d}.semantic_id.png")
    if ref_size:
        write_control_videos(idx, ep, res, m0, held, cond, ref_size, ref_t, frame0_hi)
    Image.fromarray(ep["frames"][0]).save(out / "first_frame_src.png")
    (out / "prompt.txt").write_text(ep["text"])
    info = dict(res["info"], text=ep["text"], meta={k: v for k, v in ep["meta"].items() if k != "cart"},
                states=res["states"][::5], K=res["cam"].K.tolist(), T_wc=res["cam"].T_wc.tolist(),
                dist=list(res["cam"].dist))
    (out / "info.json").write_text(json.dumps(info, indent=1, default=float))
    print("done", out, flush=True)


def write_control_videos(idx, ep, res, m0, held, cond, ref_size, ref_t, frame0_hi=None):
    """LTX control videos at W x H from the raw-resolution renders: ref_WxH.mp4 (control_frame: inverse depth, robot,
    held object) and the RGBD proxy ref_rgb_WxH.mp4 (shaded robot + held objects over the static first frame) +
    ref_depth_WxH.mp4 (inverse depth). frame0_hi (config.cfg.hires): the checked full-resolution first frame; the
    static scene is then that frame and the first-frame object masks are the full-resolution ones."""
    from simpwm.video.dataset import (
        CWM_OBJ,
        ROBOT,
        control_frame,
        depth_frame,
        first_frame_hole,
        raw_first_frame,
        world_rgb_frame,
    )

    W, H = (int(x) for x in ref_size.split("x"))
    ref = [control_frame(res["frames_hi"][t][0], raster.to_cwm(res["frames_hi"][t][1]), (W, H), res["shade_hi"].get(t))
           for t in ref_t]
    imageio.mimwrite(cond / f"ref_{ref_size}.mp4", ref, fps=ep["fps"], quality=9, macro_block_size=1)
    if not res["shade_hi"]:
        return
    if frame0_hi is not None:  # no fallback
        first = frame0_hi
    else:
        first = raw_first_frame(idx, ep)[0]  # the static scene = the checked first frame at the raw recording size
    sems = {t: raster.to_cwm(res["frames_hi"][t][1]) for t in set(ref_t.tolist())}
    zero = np.zeros(RAW_HW, bool)
    if frame0_hi is not None:
        mhi = {o.id: m for o, m in zip(res["objects0"], res["dbg"]["obj_masks_hi"])}
        sam0 = [mhi[i] for i in held if i in mhi]
    else:
        sam0 = [cv2.resize(m0[i].astype(np.uint8), RAW_HW[::-1], interpolation=cv2.INTER_NEAREST) > 0
                for i in held if i in m0]
    held0 = sems[0] == CWM_OBJ
    hole = first_frame_hole(sems[0] == ROBOT, held0)  # the proxy's own t = 0 footprint (what is drawn there)
    # the first-frame object mask beyond it: marked once a held object moved
    hole_late = first_frame_hole(zero, np.any(sam0, 0) if sam0 else zero) & ~hole
    c0 = np.array(np.nonzero(held0)).mean(1) if held0.any() else None

    def moved(sem):  # a held object's footprint centre is >= 1.5% of the width from where it started
        h = sem == CWM_OBJ
        return c0 is not None and h.any() and np.abs(np.array(np.nonzero(h)).mean(1) - c0).max() >= 0.015 * RAW_HW[1]

    imageio.mimwrite(cond / f"ref_rgb_{ref_size}.mp4",
                     [world_rgb_frame(first, res["shade_hi"][t], (sems[t] == ROBOT) | (sems[t] == CWM_OBJ), hole,
                                      (W, H), hole_late=hole_late, late=moved(sems[t])) for t in ref_t],
                     fps=ep["fps"], quality=9, macro_block_size=1)
    imageio.mimwrite(cond / f"ref_depth_{ref_size}.mp4", [depth_frame(res["frames_hi"][t][0], (W, H)) for t in ref_t],
                     fps=ep["fps"], quality=9, macro_block_size=1)


def run_all(out_dir, idxs, setups=None, stereo_dir=None, **kw):
    """`run` over a list of episodes: finished ones (info.json) and, with stereo_dir, those without stereo depth are
    skipped; a failing episode is reported and the list continues. setups: {idx: [other episodes of its setup]}."""
    ok = setup_calib.ok_entries()
    for idx in idxs:
        if (Path(out_dir) / f"droid_{idx:05d}" / "info.json").exists():
            print("skip (done)", idx, flush=True)
            continue
        if stereo_dir:
            f = Path(stereo_dir) / f"{idx}_ext1.npy"
            if not f.exists():
                print("skip (no stereo depth)", idx, flush=True)
                continue
            init_state.STEREO = np.load(f).astype(np.float32)
        try:
            run(idx, out_dir, None if setups is None else setups[str(idx)], ok, **kw)
        except Exception:  # one bad episode (e.g. a camera so far off that no support is found) must not end the list
            print(f"FAILED {idx}\n{traceback.format_exc()}", flush=True)
        finally:
            init_state.STEREO = None


def add_arguments(ap):
    """The runner's command-line options (shared with droid/run_counterfactual.py)."""
    ap.add_argument("out_dir", type=Path, help="output root (one droid_XXXXX/ directory per episode)")
    ap.add_argument("idxs", type=int, nargs="+", metavar="IDX",
                    help="episode indices (droid/data.py manifest); @FILE reads them from a file")
    ap.add_argument("--camera-dir", type=Path,
                    help="per-episode cameras <idx>.json (droid/episode_calib.py, e.g. resources/droid/cameras); "
                         "episodes without one use their setup camera")
    ap.add_argument("--logged-camera", action="store_true",
                    help="ablation: the camera DROID logged instead of the setup camera (same as SIMPWM_LOGGED_CAMERA=1)")
    ap.add_argument("--hires", action="store_true",
                    help='preset "droid_hires": 640x360 simulation, full-resolution first frame for SAM 3 and the '
                         "object textures, jaw-volume grasps, part objects, carried children (needs --ref-size and the "
                         "raw DROID source)")
    ap.add_argument("--stereo-dir", type=Path,
                    help="first-frame stereo depth <idx>_ext1.npy (droid/stereo/extract.py); episodes without one are "
                         "skipped. Default: monocular depth")
    ap.add_argument("--ref-size", metavar="WxH",
                    help="also write the LTX control videos conditions/ref*_WxH.mp4 (e.g. 1280x704)")
    ap.add_argument("--clip-frames", type=int, default=97, help="recorded steps of the LTX clip (default 97)")
    ap.add_argument("--clip-only", action="store_true",
                    help="simulate only the clip and skip the evaluation against the real video (training proxies)")
    ap.add_argument("--setups", type=Path,
                    help="JSON {idx: [other episodes of its setup]} for setup calibration (default: the setup index)")
    ap.add_argument("--vlm-dir", type=Path, default=paths.RUNS / "vlm_objects",
                    help="perception/vlm_objects.py outputs <idx>.json naming deformable objects (used when present)")


def run_from_args(args, edit=None):
    if args.hires:
        config.use("droid_hires")
    if args.logged_camera:
        config.update(logged_camera=True)
    setups = json.loads(args.setups.read_text()) if args.setups else None
    run_all(args.out_dir, args.idxs, setups, args.stereo_dir, camera_dir=args.camera_dir, ref_size=args.ref_size,
            clip_frames=args.clip_frames, clip_only=args.clip_only, vlm_dir=args.vlm_dir, edit=edit)


def main():
    ap = argparse.ArgumentParser(prog="python -m simpwm.droid.run", description=__doc__.split("\n\n")[0],
                                 fromfile_prefix_chars="@")
    ap.convert_arg_line_to_args = str.split  # @FILE: a whitespace-separated id list
    add_arguments(ap)
    run_from_args(ap.parse_args())


if __name__ == "__main__":
    main()
